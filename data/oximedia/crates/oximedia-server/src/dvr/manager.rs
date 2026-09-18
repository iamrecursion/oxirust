//! DVR manager for coordinating time-shift operations.

use crate::dvr::{DvrBuffer, DvrConfig, DvrStorage};
use crate::error::{ServerError, ServerResult};
use oximedia_net::rtmp::MediaPacket;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Time-shift request.
#[derive(Debug, Clone)]
pub struct TimeShiftRequest {
    /// Stream key.
    pub stream_key: String,

    /// Start time (milliseconds).
    pub start_time: u64,

    /// End time (milliseconds, None = live).
    pub end_time: Option<u64>,
}

/// DVR manager.
pub struct DvrManager {
    /// Configuration.
    config: DvrConfig,

    /// DVR buffers per stream.
    buffers: Arc<RwLock<HashMap<String, Arc<DvrBuffer>>>>,

    /// DVR storage.
    storage: Arc<DvrStorage>,
}

impl DvrManager {
    /// Creates a new DVR manager.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn new(config: DvrConfig, storage_dir: &str) -> ServerResult<Self> {
        let storage = Arc::new(DvrStorage::new(storage_dir)?);

        Ok(Self {
            config,
            buffers: Arc::new(RwLock::new(HashMap::new())),
            storage,
        })
    }

    /// Creates a DVR buffer for a stream.
    pub fn create_buffer(&self, stream_key: impl Into<String>) -> Arc<DvrBuffer> {
        let stream_key = stream_key.into();
        let buffer = Arc::new(DvrBuffer::new(self.config.clone()));

        let mut buffers = self.buffers.write();
        buffers.insert(stream_key, Arc::clone(&buffer));

        buffer
    }

    /// Gets a DVR buffer for a stream.
    #[must_use]
    pub fn get_buffer(&self, stream_key: &str) -> Option<Arc<DvrBuffer>> {
        let buffers = self.buffers.read();
        buffers.get(stream_key).cloned()
    }

    /// Removes a DVR buffer.
    pub fn remove_buffer(&self, stream_key: &str) {
        let mut buffers = self.buffers.write();
        buffers.remove(stream_key);
    }

    /// Handles a time-shift request.
    pub async fn handle_timeshift_request(
        &self,
        request: TimeShiftRequest,
    ) -> ServerResult<Vec<MediaPacket>> {
        let buffer = self
            .get_buffer(&request.stream_key)
            .ok_or_else(|| ServerError::NotFound("Stream not found".to_string()))?;

        // Get packets from buffer
        let packets = if let Some(end_time) = request.end_time {
            buffer.get_segments(request.start_time, end_time)
        } else {
            // Live - get from start_time to current
            if let Some((_, current_time)) = buffer.time_range() {
                buffer.get_segments(request.start_time, current_time)
            } else {
                Vec::new()
            }
        };

        Ok(packets)
    }

    /// Persists DVR buffer to storage.
    pub async fn persist_buffer(&self, stream_key: &str) -> ServerResult<()> {
        let buffer = self
            .get_buffer(stream_key)
            .ok_or_else(|| ServerError::NotFound("Stream not found".to_string()))?;

        let packets = buffer.get_all_segments();

        if !packets.is_empty() {
            self.storage.write_segment(stream_key, 0, &packets).await?;
        }

        Ok(())
    }

    /// Cleans up old DVR data.
    pub async fn cleanup(&self, stream_key: &str, keep_count: usize) -> ServerResult<()> {
        self.storage
            .cleanup_old_segments(stream_key, keep_count)
            .await
    }
}
