//! Thumbnail generation for live streams.
//!
//! This module generates preview thumbnails from live video streams,
//! useful for:
//! - Video preview in player UI
//! - Thumbnail strips for seeking
//! - Social media sharing

use super::MediaPacket;
use bytes::Bytes;
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

/// Thumbnail configuration.
#[derive(Debug, Clone)]
pub struct ThumbnailConfig {
    /// Thumbnail generation interval.
    pub interval: Duration,

    /// Thumbnail width.
    pub width: u32,

    /// Thumbnail height.
    pub height: u32,

    /// JPEG quality (1-100).
    pub quality: u8,

    /// Maximum thumbnails to keep.
    pub max_thumbnails: usize,
}

impl Default for ThumbnailConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(10),
            width: 160,
            height: 90,
            quality: 80,
            max_thumbnails: 100,
        }
    }
}

/// Generated thumbnail.
#[derive(Debug, Clone)]
pub struct Thumbnail {
    /// Timestamp of the frame (milliseconds).
    pub timestamp: u64,

    /// Thumbnail data (JPEG).
    pub data: Bytes,

    /// Width in pixels.
    pub width: u32,

    /// Height in pixels.
    pub height: u32,
}

impl Thumbnail {
    /// Creates a new thumbnail.
    #[must_use]
    pub fn new(timestamp: u64, data: Bytes, width: u32, height: u32) -> Self {
        Self {
            timestamp,
            data,
            width,
            height,
        }
    }

    /// Returns the size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

/// Thumbnail generator.
pub struct ThumbnailGenerator {
    /// Configuration.
    config: ThumbnailConfig,

    /// Generated thumbnails.
    thumbnails: RwLock<VecDeque<Arc<Thumbnail>>>,

    /// Last generation timestamp.
    last_generation_ts: RwLock<u64>,
}

impl ThumbnailGenerator {
    /// Creates a new thumbnail generator.
    #[must_use]
    pub fn new(interval: Duration) -> Self {
        Self {
            config: ThumbnailConfig {
                interval,
                ..Default::default()
            },
            thumbnails: RwLock::new(VecDeque::new()),
            last_generation_ts: RwLock::new(0),
        }
    }

    /// Creates a new thumbnail generator with config.
    #[must_use]
    pub fn with_config(config: ThumbnailConfig) -> Self {
        Self {
            config,
            thumbnails: RwLock::new(VecDeque::new()),
            last_generation_ts: RwLock::new(0),
        }
    }

    /// Generates a thumbnail from a media packet.
    pub fn generate_from_packet(&self, packet: &MediaPacket) {
        let last_ts = *self.last_generation_ts.read();
        let interval_ms = self.config.interval.as_millis() as u64;

        // Check if enough time has passed
        if packet.timestamp.saturating_sub(last_ts) < interval_ms {
            return;
        }

        // Generate thumbnail (simplified - in production would decode frame and resize)
        let thumbnail_data = Self::extract_thumbnail(
            &packet.data,
            self.config.width,
            self.config.height,
            self.config.quality,
        );

        let thumbnail = Arc::new(Thumbnail::new(
            packet.timestamp,
            thumbnail_data,
            self.config.width,
            self.config.height,
        ));

        // Store thumbnail
        {
            let mut thumbnails = self.thumbnails.write();
            thumbnails.push_back(thumbnail);

            if thumbnails.len() > self.config.max_thumbnails {
                thumbnails.pop_front();
            }
        }

        // Update last generation timestamp
        *self.last_generation_ts.write() = packet.timestamp;
    }

    /// Extracts a thumbnail from video data.
    fn extract_thumbnail(data: &[u8], _width: u32, _height: u32, _quality: u8) -> Bytes {
        // In production, this would:
        // 1. Decode the video frame
        // 2. Resize to thumbnail dimensions
        // 3. Encode as JPEG with specified quality
        //
        // For now, we'll return a placeholder
        Bytes::copy_from_slice(&data[..data.len().min(1024)])
    }

    /// Gets all thumbnails.
    #[must_use]
    pub fn get_all(&self) -> Vec<Arc<Thumbnail>> {
        let thumbnails = self.thumbnails.read();
        thumbnails.iter().cloned().collect()
    }

    /// Gets thumbnails in a time range.
    #[must_use]
    pub fn get_range(&self, start_ts: u64, end_ts: u64) -> Vec<Arc<Thumbnail>> {
        let thumbnails = self.thumbnails.read();

        thumbnails
            .iter()
            .filter(|t| t.timestamp >= start_ts && t.timestamp <= end_ts)
            .cloned()
            .collect()
    }

    /// Gets the thumbnail closest to a timestamp.
    #[must_use]
    pub fn get_closest(&self, timestamp: u64) -> Option<Arc<Thumbnail>> {
        let thumbnails = self.thumbnails.read();

        thumbnails
            .iter()
            .min_by_key(|t| {
                if t.timestamp > timestamp {
                    t.timestamp - timestamp
                } else {
                    timestamp - t.timestamp
                }
            })
            .cloned()
    }

    /// Gets the most recent thumbnail.
    #[must_use]
    pub fn get_latest(&self) -> Option<Arc<Thumbnail>> {
        let thumbnails = self.thumbnails.read();
        thumbnails.back().cloned()
    }

    /// Gets thumbnail count.
    #[must_use]
    pub fn count(&self) -> usize {
        let thumbnails = self.thumbnails.read();
        thumbnails.len()
    }

    /// Clears all thumbnails.
    pub fn clear(&self) {
        let mut thumbnails = self.thumbnails.write();
        thumbnails.clear();
        *self.last_generation_ts.write() = 0;
    }

    /// Generates a thumbnail strip (sprite sheet).
    #[must_use]
    pub fn generate_strip(&self, count: usize) -> Option<ThumbnailStrip> {
        let thumbnails = self.thumbnails.read();

        if thumbnails.is_empty() {
            return None;
        }

        // Select evenly spaced thumbnails
        let step = thumbnails.len() / count.min(thumbnails.len());
        let selected: Vec<_> = thumbnails
            .iter()
            .step_by(step.max(1))
            .take(count)
            .cloned()
            .collect();

        Some(ThumbnailStrip {
            thumbnails: selected,
            width: self.config.width,
            height: self.config.height,
        })
    }
}

/// Thumbnail strip (sprite sheet).
#[derive(Debug, Clone)]
pub struct ThumbnailStrip {
    /// Thumbnails in the strip.
    pub thumbnails: Vec<Arc<Thumbnail>>,

    /// Individual thumbnail width.
    pub width: u32,

    /// Individual thumbnail height.
    pub height: u32,
}

impl ThumbnailStrip {
    /// Generates a combined sprite sheet image.
    #[must_use]
    pub fn generate_sprite(&self) -> Bytes {
        // In production, this would combine thumbnails into a single image
        // arranged in a grid for efficient loading
        //
        // For now, return placeholder
        Bytes::new()
    }

    /// Returns total width of sprite.
    #[must_use]
    pub fn sprite_width(&self) -> u32 {
        // Assuming horizontal layout
        self.width * self.thumbnails.len() as u32
    }

    /// Returns total height of sprite.
    #[must_use]
    pub const fn sprite_height(&self) -> u32 {
        self.height
    }

    /// Gets thumbnail coordinates in sprite.
    #[must_use]
    pub fn thumbnail_coords(&self, index: usize) -> Option<(u32, u32)> {
        if index < self.thumbnails.len() {
            Some((self.width * index as u32, 0))
        } else {
            None
        }
    }
}
