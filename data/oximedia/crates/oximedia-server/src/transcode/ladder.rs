//! ABR (Adaptive Bitrate) ladder configuration.

use serde::{Deserialize, Serialize};

/// Quality level for ABR streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityLevel {
    /// Quality name (e.g., "1080p", "720p", "480p").
    pub name: String,

    /// Width in pixels.
    pub width: u32,

    /// Height in pixels.
    pub height: u32,

    /// Video bitrate in bits per second.
    pub video_bitrate: u64,

    /// Audio bitrate in bits per second.
    pub audio_bitrate: u64,

    /// Frame rate.
    pub framerate: f64,

    /// Video codec.
    pub video_codec: String,

    /// Audio codec.
    pub audio_codec: String,

    /// GOP size (Group of Pictures).
    pub gop_size: u32,

    /// Keyframe interval in seconds.
    pub keyframe_interval: f64,
}

impl QualityLevel {
    /// Creates a new quality level.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        width: u32,
        height: u32,
        video_bitrate: u64,
        audio_bitrate: u64,
    ) -> Self {
        Self {
            name: name.into(),
            width,
            height,
            video_bitrate,
            audio_bitrate,
            framerate: 30.0,
            video_codec: "av1".to_string(),  // Patent-free
            audio_codec: "opus".to_string(), // Patent-free
            gop_size: 60,
            keyframe_interval: 2.0,
        }
    }

    /// Gets total bitrate.
    #[must_use]
    pub const fn total_bitrate(&self) -> u64 {
        self.video_bitrate + self.audio_bitrate
    }

    /// Gets bandwidth (with 20% overhead for protocol).
    #[must_use]
    pub fn bandwidth(&self) -> u64 {
        (self.total_bitrate() as f64 * 1.2) as u64
    }
}

/// ABR ladder configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbrLadder {
    /// Quality levels.
    pub levels: Vec<QualityLevel>,

    /// Minimum buffer length in seconds.
    pub min_buffer_length: f64,

    /// Target buffer length in seconds.
    pub target_buffer_length: f64,

    /// Maximum buffer length in seconds.
    pub max_buffer_length: f64,
}

impl AbrLadder {
    /// Creates a new ABR ladder.
    #[must_use]
    pub fn new(levels: Vec<QualityLevel>) -> Self {
        Self {
            levels,
            min_buffer_length: 2.0,
            target_buffer_length: 6.0,
            max_buffer_length: 30.0,
        }
    }

    /// Creates a standard ABR ladder for live streaming.
    #[must_use]
    pub fn standard() -> Self {
        let levels = vec![
            QualityLevel::new("1080p", 1920, 1080, 4_500_000, 128_000),
            QualityLevel::new("720p", 1280, 720, 2_500_000, 128_000),
            QualityLevel::new("480p", 854, 480, 1_400_000, 96_000),
            QualityLevel::new("360p", 640, 360, 800_000, 96_000),
            QualityLevel::new("240p", 426, 240, 400_000, 64_000),
        ];

        Self::new(levels)
    }

    /// Creates a high-quality ABR ladder.
    #[must_use]
    pub fn high_quality() -> Self {
        let levels = vec![
            QualityLevel::new("1440p", 2560, 1440, 8_000_000, 192_000),
            QualityLevel::new("1080p", 1920, 1080, 6_000_000, 192_000),
            QualityLevel::new("720p", 1280, 720, 3_500_000, 128_000),
            QualityLevel::new("480p", 854, 480, 2_000_000, 128_000),
            QualityLevel::new("360p", 640, 360, 1_000_000, 96_000),
        ];

        Self::new(levels)
    }

    /// Creates a low-latency ABR ladder.
    #[must_use]
    pub fn low_latency() -> Self {
        let mut ladder = Self::standard();
        ladder.min_buffer_length = 0.5;
        ladder.target_buffer_length = 1.5;
        ladder.max_buffer_length = 6.0;

        // Reduce GOP size for lower latency
        for level in &mut ladder.levels {
            level.gop_size = 30;
            level.keyframe_interval = 1.0;
        }

        ladder
    }

    /// Gets a quality level by name.
    #[must_use]
    pub fn get_level(&self, name: &str) -> Option<&QualityLevel> {
        self.levels.iter().find(|l| l.name == name)
    }

    /// Gets the highest quality level.
    #[must_use]
    pub fn highest_quality(&self) -> Option<&QualityLevel> {
        self.levels.iter().max_by_key(|l| l.total_bitrate())
    }

    /// Gets the lowest quality level.
    #[must_use]
    pub fn lowest_quality(&self) -> Option<&QualityLevel> {
        self.levels.iter().min_by_key(|l| l.total_bitrate())
    }

    /// Sorts levels by bitrate (descending).
    pub fn sort_by_bitrate(&mut self) {
        self.levels
            .sort_by(|a, b| b.total_bitrate().cmp(&a.total_bitrate()));
    }
}

impl Default for AbrLadder {
    fn default() -> Self {
        Self::standard()
    }
}
