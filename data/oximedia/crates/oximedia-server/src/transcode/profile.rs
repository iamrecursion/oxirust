//! Transcoding profiles for different use cases.

use serde::{Deserialize, Serialize};

/// Video encoding profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoProfile {
    /// Codec name.
    pub codec: String,

    /// Width in pixels.
    pub width: u32,

    /// Height in pixels.
    pub height: u32,

    /// Bitrate in bits per second.
    pub bitrate: u64,

    /// Frame rate.
    pub framerate: f64,

    /// GOP size.
    pub gop_size: u32,

    /// Encoding preset (e.g., "fast", "medium", "slow").
    pub preset: String,

    /// CRF (Constant Rate Factor) for quality-based encoding.
    pub crf: Option<u8>,

    /// Number of B-frames.
    pub bframes: u32,

    /// Reference frames.
    pub ref_frames: u32,
}

impl VideoProfile {
    /// Creates a new video profile.
    #[must_use]
    pub fn new(codec: impl Into<String>, width: u32, height: u32, bitrate: u64) -> Self {
        Self {
            codec: codec.into(),
            width,
            height,
            bitrate,
            framerate: 30.0,
            gop_size: 60,
            preset: "medium".to_string(),
            crf: None,
            bframes: 3,
            ref_frames: 3,
        }
    }

    /// Creates an AV1 profile.
    #[must_use]
    pub fn av1(width: u32, height: u32, bitrate: u64) -> Self {
        Self::new("av1", width, height, bitrate)
    }

    /// Creates a VP9 profile.
    #[must_use]
    pub fn vp9(width: u32, height: u32, bitrate: u64) -> Self {
        Self::new("vp9", width, height, bitrate)
    }
}

/// Audio encoding profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioProfile {
    /// Codec name.
    pub codec: String,

    /// Bitrate in bits per second.
    pub bitrate: u64,

    /// Sample rate in Hz.
    pub sample_rate: u32,

    /// Number of channels.
    pub channels: u8,

    /// Audio encoding quality (0-10, higher is better).
    pub quality: Option<u8>,
}

impl AudioProfile {
    /// Creates a new audio profile.
    #[must_use]
    pub fn new(codec: impl Into<String>, bitrate: u64, sample_rate: u32, channels: u8) -> Self {
        Self {
            codec: codec.into(),
            bitrate,
            sample_rate,
            channels,
            quality: None,
        }
    }

    /// Creates an Opus profile.
    #[must_use]
    pub fn opus(bitrate: u64) -> Self {
        Self::new("opus", bitrate, 48000, 2)
    }

    /// Creates a Vorbis profile.
    #[must_use]
    pub fn vorbis(bitrate: u64) -> Self {
        Self::new("vorbis", bitrate, 48000, 2)
    }
}

/// Complete transcoding profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeProfile {
    /// Profile name.
    pub name: String,

    /// Video profile.
    pub video: VideoProfile,

    /// Audio profile.
    pub audio: AudioProfile,

    /// Container format.
    pub container: String,

    /// Segment duration (for HLS/DASH).
    pub segment_duration: f64,
}

impl TranscodeProfile {
    /// Creates a new transcode profile.
    #[must_use]
    pub fn new(name: impl Into<String>, video: VideoProfile, audio: AudioProfile) -> Self {
        Self {
            name: name.into(),
            video,
            audio,
            container: "webm".to_string(), // Patent-free
            segment_duration: 2.0,
        }
    }

    /// Creates a 1080p profile.
    #[must_use]
    pub fn p1080() -> Self {
        Self::new(
            "1080p",
            VideoProfile::av1(1920, 1080, 4_500_000),
            AudioProfile::opus(128_000),
        )
    }

    /// Creates a 720p profile.
    #[must_use]
    pub fn p720() -> Self {
        Self::new(
            "720p",
            VideoProfile::av1(1280, 720, 2_500_000),
            AudioProfile::opus(128_000),
        )
    }

    /// Creates a 480p profile.
    #[must_use]
    pub fn p480() -> Self {
        Self::new(
            "480p",
            VideoProfile::av1(854, 480, 1_400_000),
            AudioProfile::opus(96_000),
        )
    }

    /// Creates a 360p profile.
    #[must_use]
    pub fn p360() -> Self {
        Self::new(
            "360p",
            VideoProfile::av1(640, 360, 800_000),
            AudioProfile::opus(96_000),
        )
    }
}
