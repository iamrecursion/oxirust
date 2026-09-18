//! RTMP publish handling.

use crate::error::{ServerError, ServerResult};
use oximedia_net::rtmp::{MediaPacket, MediaPacketType, StreamMetadata};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::warn;

/// Publish context.
#[derive(Debug, Clone)]
pub struct PublishContext {
    /// Application name.
    pub app_name: String,

    /// Stream key.
    pub stream_key: String,

    /// Stream metadata.
    pub metadata: StreamMetadata,

    /// Publish start time.
    pub start_time: std::time::Instant,
}

impl PublishContext {
    /// Creates a new publish context.
    #[must_use]
    pub fn new(app_name: String, stream_key: String, metadata: StreamMetadata) -> Self {
        Self {
            app_name,
            stream_key,
            metadata,
            start_time: std::time::Instant::now(),
        }
    }

    /// Gets the stream key path.
    #[must_use]
    pub fn key_path(&self) -> String {
        format!("{}/{}", self.app_name, self.stream_key)
    }

    /// Gets publish duration.
    #[must_use]
    pub fn duration(&self) -> std::time::Duration {
        std::time::Instant::now().duration_since(self.start_time)
    }
}

/// Publish handler.
pub struct PublishHandler {
    /// Publish context.
    context: Arc<PublishContext>,

    /// Media packet sender.
    packet_tx: mpsc::UnboundedSender<MediaPacket>,

    /// Total bytes published.
    bytes_published: Arc<parking_lot::RwLock<u64>>,

    /// Total packets published.
    packets_published: Arc<parking_lot::RwLock<u64>>,

    /// Video packet count.
    video_packets: Arc<parking_lot::RwLock<u64>>,

    /// Audio packet count.
    audio_packets: Arc<parking_lot::RwLock<u64>>,

    /// Metadata packet count.
    metadata_packets: Arc<parking_lot::RwLock<u64>>,

    /// Last video timestamp.
    last_video_ts: Arc<parking_lot::RwLock<Option<u64>>>,

    /// Last audio timestamp.
    last_audio_ts: Arc<parking_lot::RwLock<Option<u64>>>,
}

impl PublishHandler {
    /// Creates a new publish handler.
    #[must_use]
    pub fn new(context: PublishContext, packet_tx: mpsc::UnboundedSender<MediaPacket>) -> Self {
        Self {
            context: Arc::new(context),
            packet_tx,
            bytes_published: Arc::new(parking_lot::RwLock::new(0)),
            packets_published: Arc::new(parking_lot::RwLock::new(0)),
            video_packets: Arc::new(parking_lot::RwLock::new(0)),
            audio_packets: Arc::new(parking_lot::RwLock::new(0)),
            metadata_packets: Arc::new(parking_lot::RwLock::new(0)),
            last_video_ts: Arc::new(parking_lot::RwLock::new(None)),
            last_audio_ts: Arc::new(parking_lot::RwLock::new(None)),
        }
    }

    /// Handles a media packet.
    pub fn handle_packet(&self, packet: MediaPacket) -> ServerResult<()> {
        // Validate packet
        self.validate_packet(&packet)?;

        // Update statistics
        let data_len = packet.data.len() as u64;
        *self.bytes_published.write() += data_len;
        *self.packets_published.write() += 1;

        match packet.packet_type {
            MediaPacketType::Video => {
                *self.video_packets.write() += 1;
                *self.last_video_ts.write() = Some(packet.timestamp as u64);
            }
            MediaPacketType::Audio => {
                *self.audio_packets.write() += 1;
                *self.last_audio_ts.write() = Some(packet.timestamp as u64);
            }
            MediaPacketType::Data => {
                *self.metadata_packets.write() += 1;
            }
        }

        // Send packet downstream
        self.packet_tx
            .send(packet)
            .map_err(|_| ServerError::Internal("Failed to send packet".to_string()))?;

        Ok(())
    }

    /// Validates a packet.
    fn validate_packet(&self, packet: &MediaPacket) -> ServerResult<()> {
        // Check for timestamp issues
        match packet.packet_type {
            MediaPacketType::Video => {
                if let Some(last_ts) = *self.last_video_ts.read() {
                    if (packet.timestamp as u64) < last_ts {
                        warn!(
                            "Video timestamp went backwards: {} -> {}",
                            last_ts, packet.timestamp
                        );
                    }
                }
            }
            MediaPacketType::Audio => {
                if let Some(last_ts) = *self.last_audio_ts.read() {
                    if (packet.timestamp as u64) < last_ts {
                        warn!(
                            "Audio timestamp went backwards: {} -> {}",
                            last_ts, packet.timestamp
                        );
                    }
                }
            }
            MediaPacketType::Data => {}
        }

        Ok(())
    }

    /// Gets publish statistics.
    #[must_use]
    pub fn get_stats(&self) -> PublishStats {
        PublishStats {
            bytes_published: *self.bytes_published.read(),
            packets_published: *self.packets_published.read(),
            video_packets: *self.video_packets.read(),
            audio_packets: *self.audio_packets.read(),
            metadata_packets: *self.metadata_packets.read(),
            last_video_ts: *self.last_video_ts.read(),
            last_audio_ts: *self.last_audio_ts.read(),
            duration: self.context.duration(),
        }
    }

    /// Gets the publish context.
    #[must_use]
    pub fn context(&self) -> &Arc<PublishContext> {
        &self.context
    }
}

/// Publish statistics.
#[derive(Debug, Clone)]
pub struct PublishStats {
    /// Total bytes published.
    pub bytes_published: u64,

    /// Total packets published.
    pub packets_published: u64,

    /// Video packet count.
    pub video_packets: u64,

    /// Audio packet count.
    pub audio_packets: u64,

    /// Metadata packet count.
    pub metadata_packets: u64,

    /// Last video timestamp.
    pub last_video_ts: Option<u64>,

    /// Last audio timestamp.
    pub last_audio_ts: Option<u64>,

    /// Publish duration.
    pub duration: std::time::Duration,
}

impl PublishStats {
    /// Gets average bitrate in bits per second.
    #[must_use]
    pub fn average_bitrate(&self) -> f64 {
        if self.duration.as_secs() > 0 {
            (self.bytes_published * 8) as f64 / self.duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Gets video bitrate in bits per second (estimated).
    #[must_use]
    pub fn video_bitrate(&self) -> f64 {
        if self.video_packets > 0 && self.duration.as_secs() > 0 {
            // Rough estimate assuming video is 80% of total
            self.average_bitrate() * 0.8
        } else {
            0.0
        }
    }

    /// Gets audio bitrate in bits per second (estimated).
    #[must_use]
    pub fn audio_bitrate(&self) -> f64 {
        if self.audio_packets > 0 && self.duration.as_secs() > 0 {
            // Rough estimate assuming audio is 20% of total
            self.average_bitrate() * 0.2
        } else {
            0.0
        }
    }
}
