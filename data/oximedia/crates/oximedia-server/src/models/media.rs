//! Media file models.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Media file metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Media {
    /// Unique media ID
    pub id: String,
    /// Owner user ID
    pub user_id: String,
    /// Stored filename
    pub filename: String,
    /// Original filename
    pub original_filename: String,
    /// MIME type
    pub mime_type: String,
    /// File size in bytes
    pub file_size: i64,
    /// Duration in seconds (for video/audio)
    pub duration: Option<f64>,
    /// Video width
    pub width: Option<i32>,
    /// Video height
    pub height: Option<i32>,
    /// Video codec
    pub codec_video: Option<String>,
    /// Audio codec
    pub codec_audio: Option<String>,
    /// Bitrate in bits/sec
    pub bitrate: Option<i64>,
    /// Framerate in fps
    pub framerate: Option<f64>,
    /// Audio sample rate
    pub sample_rate: Option<i32>,
    /// Audio channels
    pub channels: Option<i32>,
    /// Thumbnail file path
    pub thumbnail_path: Option<String>,
    /// Sprite sheet path
    pub sprite_path: Option<String>,
    /// Preview video path
    pub preview_path: Option<String>,
    /// Processing status
    pub status: MediaStatus,
    /// Creation timestamp
    pub created_at: i64,
    /// Last update timestamp
    pub updated_at: i64,
}

impl Media {
    /// Creates a new media entry.
    #[must_use]
    pub fn new(
        user_id: String,
        filename: String,
        original_filename: String,
        mime_type: String,
        file_size: i64,
    ) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: Uuid::new_v4().to_string(),
            user_id,
            filename,
            original_filename,
            mime_type,
            file_size,
            duration: None,
            width: None,
            height: None,
            codec_video: None,
            codec_audio: None,
            bitrate: None,
            framerate: None,
            sample_rate: None,
            channels: None,
            thumbnail_path: None,
            sprite_path: None,
            preview_path: None,
            status: MediaStatus::Pending,
            created_at: now,
            updated_at: now,
        }
    }

    /// Checks if this is a video file.
    #[must_use]
    pub fn is_video(&self) -> bool {
        self.mime_type.starts_with("video/")
    }

    /// Checks if this is an audio file.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        self.mime_type.starts_with("audio/")
    }

    /// Checks if this is an image file.
    #[must_use]
    pub fn is_image(&self) -> bool {
        self.mime_type.starts_with("image/")
    }
}

/// Media processing status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaStatus {
    /// Pending metadata extraction
    Pending,
    /// Processing metadata
    Processing,
    /// Ready for streaming
    Ready,
    /// Processing failed
    Failed,
}

impl std::str::FromStr for MediaStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "pending" => Ok(Self::Pending),
            "processing" => Ok(Self::Processing),
            "ready" => Ok(Self::Ready),
            "failed" => Ok(Self::Failed),
            _ => Err(format!("Invalid media status: {s}")),
        }
    }
}

impl std::fmt::Display for MediaStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Processing => write!(f, "processing"),
            Self::Ready => write!(f, "ready"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

/// Flexible metadata storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaMetadata {
    /// Media ID
    pub media_id: String,
    /// Metadata key-value pairs
    pub metadata: HashMap<String, String>,
}

impl MediaMetadata {
    /// Creates new metadata.
    #[must_use]
    pub fn new(media_id: String) -> Self {
        Self {
            media_id,
            metadata: HashMap::new(),
        }
    }

    /// Adds a metadata entry.
    pub fn add(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Gets a metadata value.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
}
