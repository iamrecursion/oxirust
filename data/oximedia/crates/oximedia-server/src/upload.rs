//! Multi-part upload handling.

use crate::{
    config::Config,
    db::Database,
    error::{ServerError, ServerResult},
    models::upload::{MultipartUpload, UploadChunk, UploadStatus},
};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncWriteExt},
};

/// Multi-part upload manager.
#[derive(Clone)]
pub struct UploadManager {
    db: Database,
    config: Config,
}

impl UploadManager {
    /// Creates a new upload manager.
    #[must_use]
    pub fn new(db: Database, config: Config) -> Self {
        Self { db, config }
    }

    /// Initializes a new multipart upload.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub async fn init_upload(
        &self,
        user_id: &str,
        filename: &str,
        total_size: i64,
        chunk_size: i64,
    ) -> ServerResult<MultipartUpload> {
        let upload = MultipartUpload::new(
            user_id.to_string(),
            filename.to_string(),
            total_size,
            chunk_size,
            24, // 24 hours expiration
        );

        crate::db::query(
            r"
            INSERT INTO multipart_uploads (
                id, user_id, filename, total_size, uploaded_size,
                chunk_size, total_chunks, completed_chunks,
                status, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ",
        )
        .bind(&upload.id)
        .bind(&upload.user_id)
        .bind(&upload.filename)
        .bind(upload.total_size)
        .bind(upload.uploaded_size)
        .bind(upload.chunk_size)
        .bind(upload.total_chunks)
        .bind(upload.completed_chunks)
        .bind(upload.status.to_string())
        .bind(upload.created_at)
        .bind(upload.expires_at)
        .execute(self.db.pool())
        .await?;

        // Create upload directory
        let upload_dir = self.config.temp_dir.join(&upload.id);
        tokio::fs::create_dir_all(&upload_dir).await?;

        Ok(upload)
    }

    /// Uploads a chunk.
    ///
    /// # Errors
    ///
    /// Returns an error if the upload fails or chunk is invalid.
    pub async fn upload_chunk(
        &self,
        upload_id: &str,
        chunk_number: i32,
        data: Vec<u8>,
    ) -> ServerResult<UploadChunk> {
        // Get upload info
        let upload = self.get_upload(upload_id).await?;

        if upload.is_expired() {
            return Err(ServerError::BadRequest("Upload expired".to_string()));
        }

        if chunk_number < 0 || chunk_number >= upload.total_chunks {
            return Err(ServerError::BadRequest(format!(
                "Invalid chunk number: {}",
                chunk_number
            )));
        }

        // Check if chunk already uploaded
        let exists = crate::db::query_scalar::<i64, _>(
            "SELECT COUNT(*) FROM upload_chunks WHERE upload_id = ? AND chunk_number = ?",
        )
        .bind(upload_id)
        .bind(chunk_number)
        .fetch_one(self.db.pool())
        .await?;

        if exists > 0 {
            return Err(ServerError::Conflict(format!(
                "Chunk {} already uploaded",
                chunk_number
            )));
        }

        // Calculate checksum
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let checksum = hex::encode(hasher.finalize());

        // Save chunk to disk
        let chunk_path = self
            .config
            .temp_dir
            .join(upload_id)
            .join(format!("chunk_{}", chunk_number));
        let mut file = File::create(&chunk_path).await?;
        file.write_all(&data).await?;

        let chunk = UploadChunk::new(
            upload_id.to_string(),
            chunk_number,
            chunk_path.to_string_lossy().to_string(),
            data.len() as i64,
            checksum,
        );

        // Save chunk info to database
        crate::db::query(
            r"
            INSERT INTO upload_chunks (
                upload_id, chunk_number, chunk_path, chunk_size, checksum, uploaded_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ",
        )
        .bind(&chunk.upload_id)
        .bind(chunk.chunk_number)
        .bind(&chunk.chunk_path)
        .bind(chunk.chunk_size)
        .bind(&chunk.checksum)
        .bind(chunk.uploaded_at)
        .execute(self.db.pool())
        .await?;

        // Update upload progress
        crate::db::query(
            r"
            UPDATE multipart_uploads
            SET completed_chunks = completed_chunks + 1,
                uploaded_size = uploaded_size + ?
            WHERE id = ?
            ",
        )
        .bind(chunk.chunk_size)
        .bind(upload_id)
        .execute(self.db.pool())
        .await?;

        Ok(chunk)
    }

    /// Completes a multipart upload by assembling all chunks.
    ///
    /// # Errors
    ///
    /// Returns an error if assembly fails or chunks are missing.
    pub async fn complete_upload(&self, upload_id: &str) -> ServerResult<PathBuf> {
        let upload = self.get_upload(upload_id).await?;

        if !upload.is_complete() {
            return Err(ServerError::BadRequest(format!(
                "Upload incomplete: {}/{} chunks",
                upload.completed_chunks, upload.total_chunks
            )));
        }

        // Update status to assembling
        crate::db::query("UPDATE multipart_uploads SET status = ? WHERE id = ?")
            .bind(UploadStatus::Assembling.to_string())
            .bind(upload_id)
            .execute(self.db.pool())
            .await?;

        // Get all chunks in order
        let rows = crate::db::query(
            r"
            SELECT chunk_path FROM upload_chunks
            WHERE upload_id = ?
            ORDER BY chunk_number
            ",
        )
        .bind(upload_id)
        .fetch_all(self.db.pool())
        .await?;

        // Assemble file
        let output_path = self.config.temp_dir.join(&upload.filename);
        let mut output_file = File::create(&output_path).await?;

        for row in rows {
            let chunk_path: String = row.get("chunk_path");
            let mut chunk_file = File::open(&chunk_path).await?;
            let mut buffer = Vec::new();
            chunk_file.read_to_end(&mut buffer).await?;
            output_file.write_all(&buffer).await?;
        }

        output_file.flush().await?;

        // Update status to completed
        crate::db::query("UPDATE multipart_uploads SET status = ? WHERE id = ?")
            .bind(UploadStatus::Completed.to_string())
            .bind(upload_id)
            .execute(self.db.pool())
            .await?;

        // Clean up chunks
        let upload_dir = self.config.temp_dir.join(upload_id);
        tokio::fs::remove_dir_all(upload_dir).await.ok();

        Ok(output_path)
    }

    /// Aborts a multipart upload.
    ///
    /// # Errors
    ///
    /// Returns an error if the abort operation fails.
    pub async fn abort_upload(&self, upload_id: &str) -> ServerResult<()> {
        // Update status
        crate::db::query("UPDATE multipart_uploads SET status = ? WHERE id = ?")
            .bind(UploadStatus::Cancelled.to_string())
            .bind(upload_id)
            .execute(self.db.pool())
            .await?;

        // Clean up chunks
        let upload_dir = self.config.temp_dir.join(upload_id);
        tokio::fs::remove_dir_all(upload_dir).await.ok();

        Ok(())
    }

    /// Gets upload information.
    ///
    /// # Errors
    ///
    /// Returns an error if the upload is not found.
    pub async fn get_upload(&self, upload_id: &str) -> ServerResult<MultipartUpload> {
        let row = crate::db::query(
            r"
            SELECT * FROM multipart_uploads WHERE id = ?
            ",
        )
        .bind(upload_id)
        .fetch_one(self.db.pool())
        .await
        .map_err(|_| ServerError::NotFound(format!("Upload not found: {}", upload_id)))?;

        let status_str: String = row.get("status");
        let status = status_str
            .parse::<UploadStatus>()
            .map_err(|e| ServerError::Internal(e))?;

        Ok(MultipartUpload {
            id: row.get("id"),
            user_id: row.get("user_id"),
            filename: row.get("filename"),
            total_size: row.get("total_size"),
            uploaded_size: row.get("uploaded_size"),
            chunk_size: row.get("chunk_size"),
            total_chunks: row.get("total_chunks"),
            completed_chunks: row.get("completed_chunks"),
            status,
            created_at: row.get("created_at"),
            expires_at: row.get("expires_at"),
        })
    }

    /// Cleans up expired uploads.
    ///
    /// # Errors
    ///
    /// Returns an error if cleanup fails.
    pub async fn cleanup_expired(&self) -> ServerResult<usize> {
        let now = chrono::Utc::now().timestamp();

        // Get expired uploads
        let rows = crate::db::query(
            r"
            SELECT id FROM multipart_uploads
            WHERE expires_at < ? AND status = 'uploading'
            ",
        )
        .bind(now)
        .fetch_all(self.db.pool())
        .await?;

        let count = rows.len();

        for row in rows {
            let upload_id: String = row.get("id");
            self.abort_upload(&upload_id).await.ok();
        }

        Ok(count)
    }
}
