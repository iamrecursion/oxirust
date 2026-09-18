//! Multi-part upload models.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Multi-part upload session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipartUpload {
    /// Unique upload ID
    pub id: String,
    /// User ID
    pub user_id: String,
    /// Filename
    pub filename: String,
    /// Total file size
    pub total_size: i64,
    /// Uploaded size so far
    pub uploaded_size: i64,
    /// Chunk size in bytes
    pub chunk_size: i64,
    /// Total number of chunks
    pub total_chunks: i32,
    /// Number of completed chunks
    pub completed_chunks: i32,
    /// Upload status
    pub status: UploadStatus,
    /// Creation timestamp
    pub created_at: i64,
    /// Expiration timestamp
    pub expires_at: i64,
}

impl MultipartUpload {
    /// Creates a new multipart upload session.
    #[must_use]
    pub fn new(
        user_id: String,
        filename: String,
        total_size: i64,
        chunk_size: i64,
        expiration_hours: i64,
    ) -> Self {
        let now = chrono::Utc::now().timestamp();
        let total_chunks = ((total_size + chunk_size - 1) / chunk_size) as i32;

        Self {
            id: Uuid::new_v4().to_string(),
            user_id,
            filename,
            total_size,
            uploaded_size: 0,
            chunk_size,
            total_chunks,
            completed_chunks: 0,
            status: UploadStatus::Uploading,
            created_at: now,
            expires_at: now + expiration_hours * 3600,
        }
    }

    /// Checks if the upload is expired.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now().timestamp() > self.expires_at
    }

    /// Checks if all chunks are uploaded.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.completed_chunks == self.total_chunks
    }
}

/// Upload chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadChunk {
    /// Upload ID
    pub upload_id: String,
    /// Chunk number (0-indexed)
    pub chunk_number: i32,
    /// Chunk file path
    pub chunk_path: String,
    /// Chunk size in bytes
    pub chunk_size: i64,
    /// Checksum (SHA-256)
    pub checksum: String,
    /// Upload timestamp
    pub uploaded_at: i64,
}

impl UploadChunk {
    /// Creates a new upload chunk.
    #[must_use]
    pub fn new(
        upload_id: String,
        chunk_number: i32,
        chunk_path: String,
        chunk_size: i64,
        checksum: String,
    ) -> Self {
        Self {
            upload_id,
            chunk_number,
            chunk_path,
            chunk_size,
            checksum,
            uploaded_at: chrono::Utc::now().timestamp(),
        }
    }
}

/// Upload status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UploadStatus {
    /// Currently uploading chunks
    Uploading,
    /// All chunks uploaded, assembling file
    Assembling,
    /// Upload completed successfully
    Completed,
    /// Upload failed
    Failed,
    /// Upload cancelled
    Cancelled,
    /// Upload expired
    Expired,
}

impl std::str::FromStr for UploadStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "uploading" => Ok(Self::Uploading),
            "assembling" => Ok(Self::Assembling),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            "expired" => Ok(Self::Expired),
            _ => Err(format!("Invalid upload status: {s}")),
        }
    }
}

impl std::fmt::Display for UploadStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uploading => write!(f, "uploading"),
            Self::Assembling => write!(f, "assembling"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::Expired => write!(f, "expired"),
        }
    }
}
