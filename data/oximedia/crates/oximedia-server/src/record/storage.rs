//! Recording storage backend.

use crate::error::{ServerError, ServerResult};
use crate::record::RecordingInfo;
use std::path::{Path, PathBuf};
use tokio::fs;
use uuid::Uuid;

/// Storage backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageBackend {
    /// Local filesystem.
    LocalFilesystem,
    /// S3-compatible storage.
    S3,
    /// Azure Blob Storage.
    Azure,
    /// Google Cloud Storage.
    Gcs,
}

/// Recording storage manager.
#[allow(dead_code)]
pub struct RecordingStorage {
    /// Storage backend.
    backend: StorageBackend,

    /// Base path for local storage.
    base_path: PathBuf,
}

impl RecordingStorage {
    /// Creates a new recording storage.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn new(base_path: impl AsRef<Path>) -> ServerResult<Self> {
        let base_path = base_path.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        std::fs::create_dir_all(&base_path)?;

        Ok(Self {
            backend: StorageBackend::LocalFilesystem,
            base_path,
        })
    }

    /// Lists all recordings.
    pub async fn list_recordings(&self) -> ServerResult<Vec<RecordingInfo>> {
        let mut recordings = Vec::new();

        let mut entries = fs::read_dir(&self.base_path).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.is_file() {
                // Try to parse recording metadata
                if let Some(info) = self.parse_recording_file(&path).await? {
                    recordings.push(info);
                }
            }
        }

        Ok(recordings)
    }

    /// Gets a recording by ID.
    pub async fn get_recording(&self, id: Uuid) -> ServerResult<Option<RecordingInfo>> {
        let recordings = self.list_recordings().await?;
        Ok(recordings.into_iter().find(|r| r.id == id))
    }

    /// Deletes a recording.
    pub async fn delete_recording(&self, id: Uuid) -> ServerResult<()> {
        if let Some(recording) = self.get_recording(id).await? {
            fs::remove_file(&recording.file_path).await?;
        }

        Ok(())
    }

    /// Parses recording file to extract metadata.
    async fn parse_recording_file(&self, path: &Path) -> ServerResult<Option<RecordingInfo>> {
        let metadata = fs::metadata(path).await?;

        // Extract information from filename
        let _filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| ServerError::Internal("Invalid filename".to_string()))?;

        // For now, create a basic recording info
        // In a real implementation, you would parse actual file metadata
        let info = RecordingInfo {
            id: Uuid::new_v4(),
            stream_key: "unknown".to_string(),
            file_path: path.to_path_buf(),
            format: crate::record::RecordingFormat::WebM,
            start_time: metadata
                .created()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .and_then(|d| chrono::DateTime::from_timestamp(d.as_secs() as i64, 0))
                .unwrap_or_else(chrono::Utc::now),
            end_time: None,
            file_size: metadata.len(),
            duration: 0,
            frames_recorded: 0,
            is_active: false,
        };

        Ok(Some(info))
    }
}
