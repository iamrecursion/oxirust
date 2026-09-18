//! Recording profiles for game capture sessions.
//!
//! A recording profile bundles encoder, audio, resolution, and quality settings
//! into a named preset that can be persisted, duplicated, and validated before
//! use.  Profiles are pure data — they carry no runtime state.

#![allow(dead_code)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Codec / Container enums
// ---------------------------------------------------------------------------

/// Video codec selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoCodec {
    /// H.264 / AVC.
    H264,
    /// H.265 / HEVC.
    H265,
    /// AV1.
    Av1,
    /// VP9.
    Vp9,
}

impl VideoCodec {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::H264 => "H.264",
            Self::H265 => "H.265",
            Self::Av1 => "AV1",
            Self::Vp9 => "VP9",
        }
    }
}

/// Audio codec selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioCodec {
    /// AAC-LC.
    Aac,
    /// Opus.
    Opus,
    /// FLAC (lossless).
    Flac,
    /// PCM (uncompressed).
    Pcm,
}

impl AudioCodec {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Aac => "AAC",
            Self::Opus => "Opus",
            Self::Flac => "FLAC",
            Self::Pcm => "PCM",
        }
    }
}

/// Container / muxer format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContainerFormat {
    /// MP4 (.mp4).
    Mp4,
    /// Matroska (.mkv).
    Mkv,
    /// `WebM` (.webm).
    Webm,
    /// MPEG-TS (.ts).
    MpegTs,
}

impl ContainerFormat {
    /// File extension (without dot).
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Mkv => "mkv",
            Self::Webm => "webm",
            Self::MpegTs => "ts",
        }
    }
}

/// Rate-control mode for the video encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControl {
    /// Constant bitrate.
    Cbr,
    /// Variable bitrate.
    Vbr,
    /// Constant quality (CRF / CQP).
    Crf,
}

// ---------------------------------------------------------------------------
// Resolution helper
// ---------------------------------------------------------------------------

/// A named resolution preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolution {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Resolution {
    /// Create a new resolution.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Total number of pixels.
    #[must_use]
    pub const fn pixels(self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Common 720p preset.
    #[must_use]
    pub const fn p720() -> Self {
        Self::new(1280, 720)
    }

    /// Common 1080p preset.
    #[must_use]
    pub const fn p1080() -> Self {
        Self::new(1920, 1080)
    }

    /// Common 1440p preset.
    #[must_use]
    pub const fn p1440() -> Self {
        Self::new(2560, 1440)
    }

    /// Common 4K preset.
    #[must_use]
    pub const fn p2160() -> Self {
        Self::new(3840, 2160)
    }

    /// Aspect ratio as a simplified string (e.g. "16:9").
    #[must_use]
    pub fn aspect_label(&self) -> String {
        let g = gcd(self.width, self.height);
        if g == 0 {
            return "0:0".to_string();
        }
        format!("{}:{}", self.width / g, self.height / g)
    }
}

/// Greatest common divisor (Euclidean).
fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

// ---------------------------------------------------------------------------
// RecordingProfile
// ---------------------------------------------------------------------------

/// A complete recording profile.
#[derive(Debug, Clone)]
pub struct RecordingProfile {
    /// Profile name (user-facing).
    pub name: String,
    /// Video codec.
    pub video_codec: VideoCodec,
    /// Audio codec.
    pub audio_codec: AudioCodec,
    /// Container format.
    pub container: ContainerFormat,
    /// Output resolution.
    pub resolution: Resolution,
    /// Target framerate.
    pub framerate: u32,
    /// Video bitrate in kbps.
    pub video_bitrate_kbps: u32,
    /// Audio bitrate in kbps.
    pub audio_bitrate_kbps: u32,
    /// Audio sample rate in Hz.
    pub audio_sample_rate: u32,
    /// Rate-control mode.
    pub rate_control: RateControl,
    /// CRF / CQP value (only used when `rate_control == Crf`).
    pub crf_value: u8,
    /// Key-frame interval in frames.
    pub keyframe_interval: u32,
    /// Arbitrary extra parameters (e.g. encoder-specific flags).
    pub extra: HashMap<String, String>,
}

impl RecordingProfile {
    /// Create a sensible "High Quality 1080p60" default profile.
    #[must_use]
    pub fn default_1080p60() -> Self {
        Self {
            name: "High Quality 1080p60".to_string(),
            video_codec: VideoCodec::H264,
            audio_codec: AudioCodec::Aac,
            container: ContainerFormat::Mp4,
            resolution: Resolution::p1080(),
            framerate: 60,
            video_bitrate_kbps: 15_000,
            audio_bitrate_kbps: 192,
            audio_sample_rate: 48_000,
            rate_control: RateControl::Vbr,
            crf_value: 18,
            keyframe_interval: 120,
            extra: HashMap::new(),
        }
    }

    /// Create a "Low-Latency Streaming" profile.
    #[must_use]
    pub fn streaming_profile() -> Self {
        Self {
            name: "Low-Latency Streaming".to_string(),
            video_codec: VideoCodec::H264,
            audio_codec: AudioCodec::Aac,
            container: ContainerFormat::MpegTs,
            resolution: Resolution::p1080(),
            framerate: 60,
            video_bitrate_kbps: 6_000,
            audio_bitrate_kbps: 128,
            audio_sample_rate: 48_000,
            rate_control: RateControl::Cbr,
            crf_value: 23,
            keyframe_interval: 120,
            extra: HashMap::new(),
        }
    }

    /// Validate the profile, returning a list of issues (empty = valid).
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.name.trim().is_empty() {
            issues.push("Profile name must not be empty".to_string());
        }
        if self.resolution.width == 0 || self.resolution.height == 0 {
            issues.push("Resolution dimensions must be non-zero".to_string());
        }
        if self.framerate == 0 || self.framerate > 240 {
            issues.push("Framerate must be in 1..=240".to_string());
        }
        if self.video_bitrate_kbps < 100 {
            issues.push("Video bitrate must be >= 100 kbps".to_string());
        }
        if self.audio_bitrate_kbps == 0 && !matches!(self.audio_codec, AudioCodec::Pcm) {
            issues.push("Audio bitrate must be > 0 for lossy codecs".to_string());
        }
        if self.keyframe_interval == 0 {
            issues.push("Keyframe interval must be > 0".to_string());
        }
        if self.container == ContainerFormat::Webm
            && !matches!(self.video_codec, VideoCodec::Vp9 | VideoCodec::Av1)
        {
            issues.push("WebM container requires VP9 or AV1 video codec".to_string());
        }

        issues
    }

    /// Whether the profile passes validation.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }

    /// Estimated output file size in megabytes for a given duration.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_size_mb(&self, duration_secs: u32) -> f64 {
        let video_mbps = f64::from(self.video_bitrate_kbps) / 1000.0;
        let audio_mbps = f64::from(self.audio_bitrate_kbps) / 1000.0;
        let total_mbps = video_mbps + audio_mbps;
        // Mbps -> MB/s  (divide by 8)
        total_mbps / 8.0 * f64::from(duration_secs)
    }

    /// Clone the profile with a new name.
    #[must_use]
    pub fn duplicate(&self, new_name: &str) -> Self {
        let mut dup = self.clone();
        dup.name = new_name.to_string();
        dup
    }
}

// ---------------------------------------------------------------------------
// ProfileLibrary
// ---------------------------------------------------------------------------

/// An in-memory collection of recording profiles.
pub struct ProfileLibrary {
    profiles: Vec<RecordingProfile>,
}

impl ProfileLibrary {
    /// Create an empty library.
    #[must_use]
    pub fn new() -> Self {
        Self {
            profiles: Vec::new(),
        }
    }

    /// Add a profile.
    pub fn add(&mut self, profile: RecordingProfile) {
        self.profiles.push(profile);
    }

    /// Number of profiles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Whether the library is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Find a profile by name.
    #[must_use]
    pub fn find(&self, name: &str) -> Option<&RecordingProfile> {
        self.profiles.iter().find(|p| p.name == name)
    }

    /// Remove a profile by name.  Returns `true` if found.
    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.profiles.len();
        self.profiles.retain(|p| p.name != name);
        self.profiles.len() < before
    }
}

impl Default for ProfileLibrary {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_codec_label() {
        assert_eq!(VideoCodec::H264.label(), "H.264");
        assert_eq!(VideoCodec::Av1.label(), "AV1");
    }

    #[test]
    fn test_audio_codec_label() {
        assert_eq!(AudioCodec::Aac.label(), "AAC");
        assert_eq!(AudioCodec::Opus.label(), "Opus");
    }

    #[test]
    fn test_container_extension() {
        assert_eq!(ContainerFormat::Mp4.extension(), "mp4");
        assert_eq!(ContainerFormat::Webm.extension(), "webm");
    }

    #[test]
    fn test_resolution_pixels() {
        let r = Resolution::p1080();
        assert_eq!(r.pixels(), 1920 * 1080);
    }

    #[test]
    fn test_resolution_aspect() {
        assert_eq!(Resolution::p1080().aspect_label(), "16:9");
        assert_eq!(Resolution::new(4, 3).aspect_label(), "4:3");
    }

    #[test]
    fn test_default_profile_valid() {
        let p = RecordingProfile::default_1080p60();
        assert!(p.is_valid());
    }

    #[test]
    fn test_streaming_profile_valid() {
        let p = RecordingProfile::streaming_profile();
        assert!(p.is_valid());
    }

    #[test]
    fn test_validation_empty_name() {
        let mut p = RecordingProfile::default_1080p60();
        p.name = String::new();
        assert!(!p.is_valid());
    }

    #[test]
    fn test_validation_zero_resolution() {
        let mut p = RecordingProfile::default_1080p60();
        p.resolution = Resolution::new(0, 0);
        assert!(!p.is_valid());
    }

    #[test]
    fn test_validation_webm_requires_vp9_or_av1() {
        let mut p = RecordingProfile::default_1080p60();
        p.container = ContainerFormat::Webm;
        p.video_codec = VideoCodec::H264;
        assert!(!p.is_valid());

        p.video_codec = VideoCodec::Vp9;
        assert!(p.is_valid());
    }

    #[test]
    fn test_estimate_size_mb() {
        let p = RecordingProfile::default_1080p60();
        let size = p.estimate_size_mb(60);
        // 15 Mbps video + 0.192 Mbps audio ≈ 15.192 Mbps ≈ 1.899 MB/s → ~114 MB
        assert!(size > 100.0 && size < 130.0);
    }

    #[test]
    fn test_duplicate_has_new_name() {
        let p = RecordingProfile::default_1080p60();
        let d = p.duplicate("Copy");
        assert_eq!(d.name, "Copy");
        assert_eq!(d.video_codec, p.video_codec);
    }

    #[test]
    fn test_library_add_find_remove() {
        let mut lib = ProfileLibrary::new();
        assert!(lib.is_empty());

        lib.add(RecordingProfile::default_1080p60());
        assert_eq!(lib.len(), 1);
        assert!(lib.find("High Quality 1080p60").is_some());

        assert!(lib.remove("High Quality 1080p60"));
        assert!(lib.is_empty());
    }

    #[test]
    fn test_library_remove_nonexistent() {
        let mut lib = ProfileLibrary::new();
        assert!(!lib.remove("nope"));
    }

    #[test]
    fn test_resolution_presets() {
        assert_eq!(Resolution::p720().width, 1280);
        assert_eq!(Resolution::p1440().width, 2560);
        assert_eq!(Resolution::p2160().width, 3840);
    }
}
