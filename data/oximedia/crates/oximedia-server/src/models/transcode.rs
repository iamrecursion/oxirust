//! Transcoding job models.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Transcoding job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeJob {
    /// Unique job ID
    pub id: String,
    /// User ID
    pub user_id: String,
    /// Source media ID
    pub media_id: String,
    /// Job status
    pub status: TranscodeStatus,
    /// Progress percentage (0.0 - 100.0)
    pub progress: f64,
    /// Output format
    pub output_format: String,
    /// Output video codec
    pub output_codec_video: Option<String>,
    /// Output audio codec
    pub output_codec_audio: Option<String>,
    /// Output width
    pub output_width: Option<i32>,
    /// Output height
    pub output_height: Option<i32>,
    /// Output bitrate
    pub output_bitrate: Option<i64>,
    /// Output file path
    pub output_path: Option<String>,
    /// Error message (if failed)
    pub error_message: Option<String>,
    /// Creation timestamp
    pub created_at: i64,
    /// Start timestamp
    pub started_at: Option<i64>,
    /// Completion timestamp
    pub completed_at: Option<i64>,
}

impl TranscodeJob {
    /// Creates a new transcoding job.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        user_id: String,
        media_id: String,
        output_format: String,
        output_codec_video: Option<String>,
        output_codec_audio: Option<String>,
        output_width: Option<i32>,
        output_height: Option<i32>,
        output_bitrate: Option<i64>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            user_id,
            media_id,
            status: TranscodeStatus::Queued,
            progress: 0.0,
            output_format,
            output_codec_video,
            output_codec_audio,
            output_width,
            output_height,
            output_bitrate,
            output_path: None,
            error_message: None,
            created_at: chrono::Utc::now().timestamp(),
            started_at: None,
            completed_at: None,
        }
    }
}

/// Transcoding job status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TranscodeStatus {
    /// Queued for processing
    Queued,
    /// Currently processing
    Processing,
    /// Completed successfully
    Completed,
    /// Failed with error
    Failed,
    /// Cancelled by user
    Cancelled,
}

impl std::str::FromStr for TranscodeStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "queued" => Ok(Self::Queued),
            "processing" => Ok(Self::Processing),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            _ => Err(format!("Invalid transcode status: {s}")),
        }
    }
}

impl std::fmt::Display for TranscodeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::Processing => write!(f, "processing"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}
