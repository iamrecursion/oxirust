//! DVR storage backend.

use crate::error::ServerResult;
use oximedia_net::rtmp::MediaPacket;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;

/// DVR segment.
#[derive(Debug, Clone)]
pub struct DvrSegment {
    /// Segment ID.
    pub id: u64,

    /// Stream key.
    pub stream_key: String,

    /// Start timestamp.
    pub start_timestamp: u64,

    /// End timestamp.
    pub end_timestamp: u64,

    /// File path.
    pub file_path: PathBuf,

    /// Size in bytes.
    pub size: u64,
}

/// DVR storage.
pub struct DvrStorage {
    /// Base directory.
    base_dir: PathBuf,
}

impl DvrStorage {
    /// Creates a new DVR storage.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn new(base_dir: impl AsRef<Path>) -> ServerResult<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_dir)?;

        Ok(Self { base_dir })
    }

    /// Writes a DVR segment to storage.
    pub async fn write_segment(
        &self,
        stream_key: &str,
        segment_id: u64,
        packets: &[MediaPacket],
    ) -> ServerResult<DvrSegment> {
        let stream_dir = self.base_dir.join(stream_key.replace('/', "_"));
        fs::create_dir_all(&stream_dir).await?;

        let filename = format!("dvr_{:08}.dat", segment_id);
        let file_path = stream_dir.join(&filename);

        // Write packets
        let mut file = fs::File::create(&file_path).await?;
        let mut total_size = 0u64;
        let mut start_ts = 0u64;
        let mut end_ts = 0u64;

        for (i, packet) in packets.iter().enumerate() {
            if i == 0 {
                start_ts = packet.timestamp as u64;
            }
            end_ts = packet.timestamp as u64;

            file.write_all(&packet.data).await?;
            total_size += packet.data.len() as u64;
        }

        file.flush().await?;

        Ok(DvrSegment {
            id: segment_id,
            stream_key: stream_key.to_string(),
            start_timestamp: start_ts,
            end_timestamp: end_ts,
            file_path,
            size: total_size,
        })
    }

    /// Reads a DVR segment from storage.
    pub async fn read_segment(&self, segment_id: u64, stream_key: &str) -> ServerResult<Vec<u8>> {
        let stream_dir = self.base_dir.join(stream_key.replace('/', "_"));
        let filename = format!("dvr_{:08}.dat", segment_id);
        let file_path = stream_dir.join(&filename);

        let data = fs::read(&file_path).await?;
        Ok(data)
    }

    /// Deletes old DVR segments.
    pub async fn cleanup_old_segments(
        &self,
        stream_key: &str,
        keep_count: usize,
    ) -> ServerResult<()> {
        let stream_dir = self.base_dir.join(stream_key.replace('/', "_"));

        if !stream_dir.exists() {
            return Ok(());
        }

        let mut entries = fs::read_dir(&stream_dir).await?;
        let mut files = Vec::new();

        while let Some(entry) = entries.next_entry().await? {
            if entry.path().extension().and_then(|s| s.to_str()) == Some("dat") {
                files.push(entry.path());
            }
        }

        // Sort by filename (segment ID)
        files.sort();

        // Delete oldest files if we have too many
        if files.len() > keep_count {
            for file in &files[..files.len() - keep_count] {
                fs::remove_file(file).await?;
            }
        }

        Ok(())
    }
}
