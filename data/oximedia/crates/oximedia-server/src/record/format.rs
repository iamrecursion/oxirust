//! Recording format definitions.

use serde::{Deserialize, Serialize};

/// Recording format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecordingFormat {
    /// WebM (VP9/Opus) - Patent-free.
    WebM,
    /// Matroska (MKV).
    Matroska,
    /// Ogg.
    Ogg,
    /// Raw FLV (for RTMP).
    Flv,
}

impl RecordingFormat {
    /// Gets the file extension for this format.
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::WebM => "webm",
            Self::Matroska => "mkv",
            Self::Ogg => "ogg",
            Self::Flv => "flv",
        }
    }

    /// Gets the MIME type for this format.
    #[must_use]
    pub const fn mime_type(&self) -> &'static str {
        match self {
            Self::WebM => "video/webm",
            Self::Matroska => "video/x-matroska",
            Self::Ogg => "video/ogg",
            Self::Flv => "video/x-flv",
        }
    }

    /// Checks if the format supports streaming.
    #[must_use]
    pub const fn supports_streaming(&self) -> bool {
        matches!(self, Self::WebM | Self::Matroska)
    }
}

impl Default for RecordingFormat {
    fn default() -> Self {
        Self::WebM
    }
}

/// Format writer trait.
pub trait FormatWriter: Send + Sync {
    /// Writes video data.
    fn write_video(&mut self, data: &[u8], timestamp: u64) -> Result<(), std::io::Error>;

    /// Writes audio data.
    fn write_audio(&mut self, data: &[u8], timestamp: u64) -> Result<(), std::io::Error>;

    /// Writes metadata.
    fn write_metadata(&mut self, data: &[u8]) -> Result<(), std::io::Error>;

    /// Finalizes the file.
    fn finalize(&mut self) -> Result<(), std::io::Error>;
}
