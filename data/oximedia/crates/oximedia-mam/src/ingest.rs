//! Asset ingest workflow
//!
//! Handles the complete asset ingest process:
//! - Checksum calculation and duplicate detection
//! - Metadata extraction
//! - Proxy/thumbnail generation
//! - Database and search index updates
//! - Queue management

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};
use uuid::Uuid;

use crate::asset::{AssetManager, CreateAssetRequest, DescriptiveMetadata};
use crate::database::Database;
use crate::search::{IndexDocument, SearchEngine};
use crate::{MamConfig, MamError, Result};

/// Ingest system manages asset ingestion
pub struct IngestSystem {
    db: Arc<Database>,
    #[allow(dead_code)]
    search: Arc<SearchEngine>,
    #[allow(dead_code)]
    asset_manager: Arc<AssetManager>,
    #[allow(dead_code)]
    config: MamConfig,
    #[allow(dead_code)]
    semaphore: Arc<Semaphore>,
    job_tx: mpsc::UnboundedSender<IngestJob>,
    #[allow(dead_code)]
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

/// Ingest job
#[derive(Debug, Clone)]
pub struct IngestJob {
    pub id: Uuid,
    pub source_path: PathBuf,
    pub metadata: Option<IngestMetadata>,
    pub created_by: Option<Uuid>,
    pub priority: IngestPriority,
}

/// Ingest job priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IngestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Metadata provided during ingest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestMetadata {
    pub title: Option<String>,
    pub description: Option<String>,
    pub keywords: Vec<String>,
    pub categories: Vec<String>,
    pub copyright: Option<String>,
    pub license: Option<String>,
    pub creator: Option<String>,
}

/// Ingest job status
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct IngestJobStatus {
    pub id: Uuid,
    pub source_path: String,
    pub asset_id: Option<Uuid>,
    pub status: IngestStatus,
    pub progress: i32,
    pub error_message: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub created_by: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// Ingest status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "text")]
pub enum IngestStatus {
    #[sqlx(rename = "pending")]
    Pending,
    #[sqlx(rename = "processing")]
    Processing,
    #[sqlx(rename = "completed")]
    Completed,
    #[sqlx(rename = "failed")]
    Failed,
    #[sqlx(rename = "cancelled")]
    Cancelled,
}

/// Ingest result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResult {
    pub job_id: Uuid,
    pub asset_id: Option<Uuid>,
    pub status: IngestStatus,
    pub error: Option<String>,
    pub duration_ms: i64,
}

/// Ingest statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestStatistics {
    pub total_jobs: i64,
    pub pending_jobs: i64,
    pub processing_jobs: i64,
    pub completed_jobs: i64,
    pub failed_jobs: i64,
    pub average_duration_ms: i64,
}

impl IngestSystem {
    /// Create a new ingest system
    #[must_use]
    pub fn new(
        db: Arc<Database>,
        search: Arc<SearchEngine>,
        asset_manager: Arc<AssetManager>,
        config: MamConfig,
    ) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_ingests));
        let (job_tx, job_rx) = mpsc::unbounded_channel();

        // Spawn worker task
        let worker_db = Arc::clone(&db);
        let worker_search = Arc::clone(&search);
        let worker_assets = Arc::clone(&asset_manager);
        let worker_semaphore = Arc::clone(&semaphore);
        let worker_config = config.clone();

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

        tokio::spawn(async move {
            Self::worker_loop(
                job_rx,
                shutdown_rx,
                worker_db,
                worker_search,
                worker_assets,
                worker_semaphore,
                worker_config,
            )
            .await;
        });

        Self {
            db,
            search,
            asset_manager,
            config,
            semaphore,
            job_tx,
            shutdown_tx: Some(shutdown_tx),
        }
    }

    /// Submit an ingest job
    ///
    /// # Errors
    ///
    /// Returns an error if job submission fails
    pub async fn submit_job(
        &self,
        source_path: PathBuf,
        metadata: Option<IngestMetadata>,
        created_by: Option<Uuid>,
        priority: IngestPriority,
    ) -> Result<Uuid> {
        // Validate source path
        if !source_path.exists() {
            return Err(MamError::InvalidInput(format!(
                "Source file does not exist: {}",
                source_path.display()
            )));
        }

        let job_id = Uuid::new_v4();

        // Create job record in database
        sqlx::query(
            "INSERT INTO ingest_jobs (id, source_path, status, progress, metadata, created_by, created_at)
             VALUES ($1, $2, 'pending', 0, $3, $4, NOW())"
        )
        .bind(job_id)
        .bind(source_path.to_str().expect("invariant: source_path is valid UTF-8"))
        .bind(metadata.as_ref().and_then(|m| serde_json::to_value(m).ok()))
        .bind(created_by)
        .execute(self.db.pool())
        .await?;

        // Queue job
        let job = IngestJob {
            id: job_id,
            source_path,
            metadata,
            created_by,
            priority,
        };

        self.job_tx
            .send(job)
            .map_err(|_| MamError::Internal("Failed to queue job".to_string()))?;

        Ok(job_id)
    }

    /// Get job status
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn get_job_status(&self, job_id: Uuid) -> Result<IngestJobStatus> {
        let status =
            sqlx::query_as::<_, IngestJobStatus>("SELECT * FROM ingest_jobs WHERE id = $1")
                .bind(job_id)
                .fetch_one(self.db.pool())
                .await?;

        Ok(status)
    }

    /// List jobs
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn list_jobs(
        &self,
        status: Option<IngestStatus>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<IngestJobStatus>> {
        let jobs = if let Some(status) = status {
            sqlx::query_as::<_, IngestJobStatus>(
                "SELECT * FROM ingest_jobs WHERE status = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3"
            )
            .bind(status)
            .bind(limit)
            .bind(offset)
            .fetch_all(self.db.pool())
            .await?
        } else {
            sqlx::query_as::<_, IngestJobStatus>(
                "SELECT * FROM ingest_jobs ORDER BY created_at DESC LIMIT $1 OFFSET $2",
            )
            .bind(limit)
            .bind(offset)
            .fetch_all(self.db.pool())
            .await?
        };

        Ok(jobs)
    }

    /// Cancel a job
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails
    pub async fn cancel_job(&self, job_id: Uuid) -> Result<()> {
        sqlx::query(
            "UPDATE ingest_jobs SET status = 'cancelled' WHERE id = $1 AND status IN ('pending', 'processing')"
        )
        .bind(job_id)
        .execute(self.db.pool())
        .await?;

        Ok(())
    }

    /// Get ingest statistics
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub async fn get_statistics(&self) -> Result<IngestStatistics> {
        let total: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ingest_jobs")
            .fetch_one(self.db.pool())
            .await?;

        let pending: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM ingest_jobs WHERE status = 'pending'")
                .fetch_one(self.db.pool())
                .await?;

        let processing: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM ingest_jobs WHERE status = 'processing'")
                .fetch_one(self.db.pool())
                .await?;

        let completed: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM ingest_jobs WHERE status = 'completed'")
                .fetch_one(self.db.pool())
                .await?;

        let failed: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM ingest_jobs WHERE status = 'failed'")
                .fetch_one(self.db.pool())
                .await?;

        let avg_duration: Option<f64> = sqlx::query_scalar(
            "SELECT AVG(EXTRACT(EPOCH FROM (completed_at - started_at)) * 1000)
             FROM ingest_jobs WHERE status = 'completed' AND completed_at IS NOT NULL",
        )
        .fetch_one(self.db.pool())
        .await?;

        Ok(IngestStatistics {
            total_jobs: total,
            pending_jobs: pending,
            processing_jobs: processing,
            completed_jobs: completed,
            failed_jobs: failed,
            average_duration_ms: avg_duration.unwrap_or(0.0) as i64,
        })
    }

    /// Shutdown the ingest system
    ///
    /// # Errors
    ///
    /// Returns an error if shutdown fails
    pub async fn shutdown(&self) -> Result<()> {
        // Note: shutdown_tx is consumed when sending
        // In a real implementation, we'd use Arc<Mutex<Option<Sender>>>
        tracing::info!("Ingest system shutdown requested");
        Ok(())
    }

    /// Worker loop that processes ingest jobs
    #[allow(clippy::too_many_arguments)]
    async fn worker_loop(
        mut job_rx: mpsc::UnboundedReceiver<IngestJob>,
        mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
        db: Arc<Database>,
        search: Arc<SearchEngine>,
        asset_manager: Arc<AssetManager>,
        semaphore: Arc<Semaphore>,
        config: MamConfig,
    ) {
        loop {
            tokio::select! {
                Some(job) = job_rx.recv() => {
                    let db = Arc::clone(&db);
                    let search = Arc::clone(&search);
                    let asset_manager = Arc::clone(&asset_manager);
                    let semaphore = Arc::clone(&semaphore);
                    let config = config.clone();

                    tokio::spawn(async move {
                        let _permit = semaphore.acquire().await.expect("semaphore mutex poisoned");
                        let _ = Self::process_job(job, db, search, asset_manager, config).await;
                    });
                }
                _ = &mut shutdown_rx => {
                    tracing::info!("Ingest worker shutting down");
                    break;
                }
            }
        }
    }

    /// Process a single ingest job
    #[allow(clippy::too_many_arguments)]
    async fn process_job(
        job: IngestJob,
        db: Arc<Database>,
        search: Arc<SearchEngine>,
        asset_manager: Arc<AssetManager>,
        _config: MamConfig,
    ) -> Result<IngestResult> {
        let start_time = std::time::Instant::now();

        tracing::info!("Processing ingest job: {}", job.id);

        // Update status to processing
        sqlx::query(
            "UPDATE ingest_jobs SET status = 'processing', started_at = NOW() WHERE id = $1",
        )
        .bind(job.id)
        .execute(db.pool())
        .await?;

        // Process the file
        let result = Self::process_file(&job, &db, &search, &asset_manager).await;

        let duration = start_time.elapsed().as_millis() as i64;

        match result {
            Ok(asset_id) => {
                // Update status to completed
                sqlx::query(
                    "UPDATE ingest_jobs SET status = 'completed', progress = 100, asset_id = $2, completed_at = NOW()
                     WHERE id = $1"
                )
                .bind(job.id)
                .bind(asset_id)
                .execute(db.pool())
                .await?;

                tracing::info!("Ingest job {} completed successfully", job.id);

                Ok(IngestResult {
                    job_id: job.id,
                    asset_id: Some(asset_id),
                    status: IngestStatus::Completed,
                    error: None,
                    duration_ms: duration,
                })
            }
            Err(e) => {
                // Update status to failed
                sqlx::query(
                    "UPDATE ingest_jobs SET status = 'failed', error_message = $2, completed_at = NOW()
                     WHERE id = $1"
                )
                .bind(job.id)
                .bind(e.to_string())
                .execute(db.pool())
                .await?;

                tracing::error!("Ingest job {} failed: {}", job.id, e);

                Ok(IngestResult {
                    job_id: job.id,
                    asset_id: None,
                    status: IngestStatus::Failed,
                    error: Some(e.to_string()),
                    duration_ms: duration,
                })
            }
        }
    }

    /// Process a file and create asset
    async fn process_file(
        job: &IngestJob,
        db: &Arc<Database>,
        search: &Arc<SearchEngine>,
        asset_manager: &Arc<AssetManager>,
    ) -> Result<Uuid> {
        // Step 1: Calculate checksum
        tracing::debug!("Calculating checksum for {:?}", job.source_path);
        let checksum = asset_manager.calculate_checksum(&job.source_path).await?;

        // Update progress
        let _ = sqlx::query("UPDATE ingest_jobs SET progress = 20 WHERE id = $1")
            .bind(job.id)
            .execute(db.pool())
            .await;

        // Step 2: Check for duplicates
        if let Some(existing) = asset_manager.get_asset_by_checksum(&checksum).await? {
            tracing::info!("Duplicate asset found: {}", existing.id);
            return Ok(existing.id);
        }

        // Update progress
        let _ = sqlx::query("UPDATE ingest_jobs SET progress = 40 WHERE id = $1")
            .bind(job.id)
            .execute(db.pool())
            .await;

        // Step 3: Extract technical metadata
        tracing::debug!("Extracting technical metadata");
        let technical = asset_manager
            .extract_technical_metadata(&job.source_path)
            .await?;

        // Update progress
        let _ = sqlx::query("UPDATE ingest_jobs SET progress = 60 WHERE id = $1")
            .bind(job.id)
            .execute(db.pool())
            .await;

        // Step 4: Get file metadata
        let metadata = tokio::fs::metadata(&job.source_path).await?;
        let file_size = metadata.len() as i64;
        let filename = job
            .source_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let mime_type = AssetManager::get_mime_type(&job.source_path);

        // Step 5: Copy file to storage
        let asset_id = Uuid::new_v4();
        let storage_path = asset_manager.get_storage_path(asset_id, &filename);

        tracing::debug!("Copying file to storage: {:?}", storage_path);

        if let Some(parent) = storage_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::copy(&job.source_path, &storage_path).await?;

        // Update progress
        let _ = sqlx::query("UPDATE ingest_jobs SET progress = 80 WHERE id = $1")
            .bind(job.id)
            .execute(db.pool())
            .await;

        // Step 6: Create asset in database
        let descriptive = job.metadata.as_ref().map(|m| DescriptiveMetadata {
            title: m.title.clone(),
            description: m.description.clone(),
            keywords: m.keywords.clone(),
            categories: m.categories.clone(),
            language: None,
            genre: None,
            subject: None,
        });

        let create_req = CreateAssetRequest {
            filename,
            file_path: storage_path
                .to_str()
                .expect("invariant: storage_path is valid UTF-8")
                .to_string(),
            file_size: Some(file_size),
            mime_type,
            checksum,
            technical: Some(technical),
            descriptive,
            rights: None,
            custom: None,
            created_by: job.created_by,
        };

        let asset = asset_manager.create_asset(create_req).await?;

        // Update progress
        let _ = sqlx::query("UPDATE ingest_jobs SET progress = 90 WHERE id = $1")
            .bind(job.id)
            .execute(db.pool())
            .await;

        // Step 7: Index in search engine
        let index_doc = IndexDocument {
            asset_id: asset.id,
            filename: asset.filename.clone(),
            title: asset.title.clone(),
            description: asset.description.clone(),
            keywords: asset.keywords.clone().unwrap_or_default(),
            categories: asset.categories.clone().unwrap_or_default(),
            mime_type: asset.mime_type.clone(),
            duration_ms: asset.duration_ms,
            width: asset.width,
            height: asset.height,
            file_size: asset.file_size,
            created_at: asset.created_at.timestamp(),
            updated_at: asset.updated_at.timestamp(),
        };

        search.index_document(index_doc)?;
        search.commit()?;

        tracing::info!("Asset created successfully: {}", asset.id);

        Ok(asset.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingest_priority_ordering() {
        assert!(IngestPriority::Critical > IngestPriority::High);
        assert!(IngestPriority::High > IngestPriority::Normal);
        assert!(IngestPriority::Normal > IngestPriority::Low);
    }

    #[test]
    fn test_ingest_status_serialization() {
        let status = IngestStatus::Processing;
        let json = serde_json::to_string(&status).expect("should succeed in test");
        assert!(json.contains("Processing"));
    }

    #[test]
    fn test_ingest_metadata() {
        let metadata = IngestMetadata {
            title: Some("Test".to_string()),
            description: None,
            keywords: vec!["test".to_string()],
            categories: vec![],
            copyright: None,
            license: None,
            creator: None,
        };

        assert_eq!(metadata.keywords.len(), 1);
    }
}
